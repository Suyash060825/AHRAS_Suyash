from __future__ import annotations
import unittest
from rbac.permissions import Perm, Role, ROLE_PERMISSIONS, role_has_permission
from auth.manager import (
    hash_password, verify_password, create_access_token, verify_token,
    authenticate_user, list_users, register_user,
)


class TestRBACAndAuth(unittest.TestCase):
    def test_01_password_hashing_and_verification(self):
        plain = "SuperSecureSOC2026!"
        h = hash_password(plain)
        self.assertTrue(verify_password(plain, h))
        self.assertFalse(verify_password("WrongPassword", h))

    def test_02_jwt_token_lifecycle(self):
        payload = {"sub": "analyst", "role": "soc_analyst"}
        token = create_access_token(payload, expires_delta_seconds=3600)
        self.assertIsNotNone(token)
        
        decoded = verify_token(token)
        self.assertIsNotNone(decoded)
        self.assertEqual(decoded["sub"], "analyst")
        self.assertEqual(decoded["role"], "soc_analyst")

    def test_03_expired_token(self):
        payload = {"sub": "analyst", "role": "soc_analyst"}
        token = create_access_token(payload, expires_delta_seconds=-10)
        decoded = verify_token(token)
        self.assertIsNone(decoded)

    def test_04_default_user_authentication(self):
        user = authenticate_user("admin", "AdminSecurePass2026!")
        self.assertIsNotNone(user)
        self.assertEqual(user["role"], "admin")
        
        bad_user = authenticate_user("admin", "WrongPass")
        self.assertIsNone(bad_user)

    def test_05_role_permission_enforcement(self):
        # Admin has all permissions
        self.assertTrue(role_has_permission("admin", Perm.SYSTEM_CONFIG))
        self.assertTrue(role_has_permission("admin", Perm.FIREWALL_CONTROL))
        
        # SOC Analyst cannot control firewall or manage users
        self.assertTrue(role_has_permission("soc_analyst", Perm.ALERTS_READ))
        self.assertFalse(role_has_permission("soc_analyst", Perm.FIREWALL_CONTROL))
        self.assertFalse(role_has_permission("soc_analyst", Perm.USERS_MANAGE))
        
        # Threat Hunter has HUNT_EXECUTE and FORENSICS
        self.assertTrue(role_has_permission("threat_hunter", Perm.HUNT_EXECUTE))
        self.assertTrue(role_has_permission("threat_hunter", Perm.TI_FEEDS_MANAGE))
        self.assertFalse(role_has_permission("threat_hunter", Perm.FIREWALL_CONTROL))
        
        # Incident Responder has SOAR and FIREWALL_CONTROL
        self.assertTrue(role_has_permission("incident_responder", Perm.SOAR_EXECUTE))
        self.assertTrue(role_has_permission("incident_responder", Perm.FIREWALL_CONTROL))
        
        # Manager is read-only reporting
        self.assertTrue(role_has_permission("manager", Perm.REPORTS_GENERATE))
        self.assertFalse(role_has_permission("manager", Perm.SOAR_EXECUTE))

    def test_06_user_registration(self):
        username = "new_responder"
        reg = register_user(username, "NewPass123!", "incident_responder", "new@ahras.security")
        self.assertEqual(reg["username"], username)
        self.assertEqual(reg["role"], "incident_responder")
        
        auth = authenticate_user(username, "NewPass123!")
        self.assertIsNotNone(auth)


if __name__ == "__main__":
    unittest.main()
